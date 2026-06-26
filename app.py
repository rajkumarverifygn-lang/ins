from __future__ import annotations

import base64
import html
import io
import importlib.metadata
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx

import config
from backend import IMAGE_EXTS, Annotation, PromptSpec, SAM3TextBackend

try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

MODEL_DOWNLOAD_LOCK = threading.Lock()

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


def render_dependency_error():
    st.error("This deployment could not load OpenCV, so image detection cannot start yet.")
    st.markdown(
        """
        Update your app dependencies with one of these options:

        - `opencv-python-headless>=4.8` for Streamlit Cloud or Linux
        - `opencv-python>=4.8` for Windows local runs
        """
    )
    if CV2_IMPORT_ERROR is not None:
        st.caption(f"OpenCV import error: {CV2_IMPORT_ERROR}")


def ensure_cv2_ready() -> bool:
    if cv2 is not None:
        return True
    render_dependency_error()
    return False


def parse_basic_toml_value(value: str) -> str:
    text = value.strip()
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text.strip()


def read_local_secrets_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        if tomllib is not None:
            with open(path, "rb") as file:
                data = tomllib.load(file)
            return {str(key): value for key, value in data.items() if not isinstance(value, dict)}
    except Exception:
        pass

    secrets: Dict[str, Any] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            secrets[key.strip()] = parse_basic_toml_value(value)
    except Exception:
        return {}
    return secrets


def local_secret_candidate_paths() -> List[Path]:
    return [
        Path(config.APP_DIR) / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]


def load_local_secrets() -> Dict[str, Any]:
    secrets: Dict[str, Any] = {}
    seen: set[str] = set()
    for path in local_secret_candidate_paths():
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        secrets.update(read_local_secrets_file(path))
    return secrets


def primary_local_secrets_path() -> Path:
    return local_secret_candidate_paths()[0]


def get_managed_streamlit_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or "").strip()


def get_streamlit_secret(name: str) -> str:
    # Check env vars first, then app-local/home secrets files for localhost,
    # then Streamlit-managed secrets for cloud deployments.
    value = os.getenv(name)
    if value:
        return str(value).strip()
    value = load_local_secrets().get(name, "")
    if value:
        return str(value).strip()
    return get_managed_streamlit_secret(name)


def get_deployment_setting(name: str, default: str = "") -> str:
    secret_value = get_streamlit_secret(name)
    if secret_value:
        return secret_value
    value = getattr(config, name, default)
    return str(value or default or "").strip()


def selected_inference_backend() -> str:
    backend_name = get_deployment_setting("INFERENCE_BACKEND", "verifygncloud").lower() or "verifygncloud"
    legacy_backend_name = config.legacy_name("VERIFYGNCLOUD").lower()
    return "verifygncloud" if backend_name == legacy_backend_name else backend_name


def get_verifygncloud_api_key() -> str:
    key_env_name = (
        get_deployment_setting("VERIFYGNCLOUD_API_KEY_ENV", "VERIFYGNCLOUD_API_KEY")
        or "VERIFYGNCLOUD_API_KEY"
    )
    api_key = os.getenv(key_env_name) or get_streamlit_secret(key_env_name)
    if not api_key and key_env_name != "VERIFYGNCLOUD_API_KEY":
        api_key = os.getenv("VERIFYGNCLOUD_API_KEY") or get_streamlit_secret("VERIFYGNCLOUD_API_KEY")
    if not api_key:
        legacy_key_name = key_env_name.replace("VERIFYGNCLOUD", "ROBOFLOW")
        api_key = os.getenv(legacy_key_name) or get_streamlit_secret(legacy_key_name)
    if not api_key:
        legacy_default_key_name = config.legacy_name("VERIFYGNCLOUD_API_KEY")
        api_key = os.getenv(legacy_default_key_name) or get_streamlit_secret(legacy_default_key_name)
    return str(api_key or "").strip()


@st.cache_resource(show_spinner=False)
def get_verifygncloud_client(api_url: str, api_key: str):
    try:
        from inference_sdk import InferenceHTTPClient
    except (ModuleNotFoundError, ImportError) as exc:
        missing = getattr(exc, "name", "inference-sdk")
        raise RuntimeError(
            f"Missing Python package `{missing}`. Add `inference-sdk` to requirements.txt, "
            "push to GitHub, and reboot the Streamlit app."
        ) from exc
    return InferenceHTTPClient(api_url=api_url, api_key=api_key)


def get_inference_sdk_version() -> str:
    try:
        return importlib.metadata.version("inference-sdk")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def extract_named_output(payload: Any, name: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if name in payload:
            return payload[name]
        for value in payload.values():
            found = extract_named_output(value, name)
            if found is not None:
                return found
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            found = extract_named_output(value, name)
            if found is not None:
                return found
    return None


def image_from_base64_text(value: str) -> Optional[np.ndarray]:
    text = value.strip()
    if not text:
        return None
    if text.startswith("data:image") and "," in text:
        text = text.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(text, validate=True)
    except Exception:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    return np.asarray(image)


def decode_image_payload(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8, copy=False)
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    if isinstance(value, dict):
        for key in ("value", "image", "base64", "data", "url", "path"):
            if key in value:
                decoded = decode_image_payload(value[key])
                if decoded is not None:
                    return decoded
        for child in value.values():
            decoded = decode_image_payload(child)
            if decoded is not None:
                return decoded
    if isinstance(value, (list, tuple)):
        for child in value:
            decoded = decode_image_payload(child)
            if decoded is not None:
                return decoded
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("http://", "https://")):
            try:
                with urlopen(text, timeout=60) as response:
                    image = Image.open(io.BytesIO(response.read())).convert("RGB")
                return np.asarray(image)
            except Exception:
                return None
        return image_from_base64_text(text)
    return None


def iter_prediction_dicts(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, dict):
        geometry_keys = {
            "x",
            "y",
            "width",
            "height",
            "bbox",
            "bounding_box",
            "points",
            "polygon",
            "segmentation",
        }
        label_keys = {"class", "class_name", "label", "name"}
        is_prediction = bool(geometry_keys.intersection(payload.keys())) or bool(
            label_keys.intersection(payload.keys()) and {"confidence", "score", "probability"}.intersection(payload.keys())
        )
        if is_prediction:
            yield payload
        skip_nested_keys = {"bbox", "bounding_box", "points", "polygon", "segmentation"} if is_prediction else set()
        for key, value in payload.items():
            if key in skip_nested_keys:
                continue
            yield from iter_prediction_dicts(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from iter_prediction_dicts(value)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def label_for_prediction(raw_label: str, prompt_to_label: Dict[str, str]) -> str:
    normalized = str(raw_label or "").strip()
    if not normalized:
        return next(iter(prompt_to_label.values()), "object")
    exact = prompt_to_label.get(normalized)
    if exact:
        return exact
    lowered = normalized.lower()
    for prompt, label in prompt_to_label.items():
        if prompt.lower() == lowered or label.lower() == lowered:
            return label
    return normalized


def points_from_prediction(value: Any) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    if not isinstance(value, (list, tuple)):
        return points
    for item in value:
        if isinstance(item, dict):
            x = item.get("x", item.get("X"))
            y = item.get("y", item.get("Y"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            continue
        points.append((int(round(to_float(x))), int(round(to_float(y)))))
    return points


def bbox_from_prediction(prediction: Dict[str, Any], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if all(key in prediction for key in ("x", "y", "width", "height")):
        cx = to_float(prediction.get("x"))
        cy = to_float(prediction.get("y"))
        bw = to_float(prediction.get("width"))
        bh = to_float(prediction.get("height"))
        x1 = clamp_int(cx - bw / 2.0, 0, width - 1)
        y1 = clamp_int(cy - bh / 2.0, 0, height - 1)
        x2 = clamp_int(cx + bw / 2.0, 0, width - 1)
        y2 = clamp_int(cy + bh / 2.0, 0, height - 1)
        return (x1, y1, max(x1, x2), max(y1, y2))

    for keys in (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax")):
        if all(key in prediction for key in keys):
            x1, y1, x2, y2 = [to_float(prediction.get(key)) for key in keys]
            return (
                clamp_int(x1, 0, width - 1),
                clamp_int(y1, 0, height - 1),
                clamp_int(x2, 0, width - 1),
                clamp_int(y2, 0, height - 1),
            )

    box = prediction.get("bbox") or prediction.get("bounding_box")
    if isinstance(box, dict):
        return bbox_from_prediction(box, width, height)
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        x1, y1, third, fourth = [to_float(v) for v in box[:4]]
        if third > x1 and fourth > y1:
            x2, y2 = third, fourth
        else:
            x2, y2 = x1 + third, y1 + fourth
        return (
            clamp_int(x1, 0, width - 1),
            clamp_int(y1, 0, height - 1),
            clamp_int(x2, 0, width - 1),
            clamp_int(y2, 0, height - 1),
        )
    return None


def mask_from_prediction(prediction: Dict[str, Any], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if not ensure_cv2_ready():
        return None
    height, width = shape_hw
    polygon_points = (
        points_from_prediction(prediction.get("points"))
        or points_from_prediction(prediction.get("polygon"))
        or points_from_prediction(prediction.get("segmentation"))
    )
    if len(polygon_points) >= 3:
        mask = np.zeros(shape_hw, dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon_points, dtype=np.int32)], 1)
        return mask

    bbox = bbox_from_prediction(prediction, width, height)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros(shape_hw, dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def annotations_from_verifygncloud_predictions(
    predictions_payload: Any,
    prompt_specs: List[PromptSpec],
    shape_hw: Tuple[int, int],
) -> List[Annotation]:
    prompt_to_label = {spec.prompt: spec.label for spec in prompt_specs}
    anns: List[Annotation] = []
    for prediction in iter_prediction_dicts(predictions_payload):
        raw_label = (
            prediction.get("class")
            or prediction.get("class_name")
            or prediction.get("label")
            or prediction.get("name")
            or ""
        )
        label = label_for_prediction(str(raw_label), prompt_to_label)
        score = (
            prediction.get("confidence")
            or prediction.get("score")
            or prediction.get("probability")
            or prediction.get("confidence_score")
            or 0.0
        )
        mask = mask_from_prediction(prediction, shape_hw)
        if mask is None or int(np.count_nonzero(mask)) <= 0:
            continue
        anns.append(Annotation.from_mask(len(anns) + 1, label, to_float(score), mask, color_index=len(anns)))
    return anns


def run_verifygncloud_workflow(image_path: Path, prompt_specs: List[PromptSpec]) -> Any:
    api_key = get_verifygncloud_api_key()
    if not api_key:
        local_hint = primary_local_secrets_path()
        raise RuntimeError(
            "VERIFYGNCLOUD API key is missing. For localhost, create "
            f"`{local_hint}` with `VERIFYGNCLOUD_API_KEY = \"your_key\"`. "
            "Do not put the real key in `secrets.toml.example`. For Streamlit Cloud, add the key in Secrets and reboot the app."
        )

    api_url = get_deployment_setting("VERIFYGNCLOUD_API_URL", "https://serverless.roboflow.com")
    workspace = get_deployment_setting("VERIFYGNCLOUD_WORKSPACE", "rajkumarm")
    workflow_id = get_deployment_setting("VERIFYGNCLOUD_WORKFLOW_ID", "general-segmentation-api")
    image_input = get_deployment_setting("VERIFYGNCLOUD_IMAGE_INPUT", "image") or "image"
    classes_input = get_deployment_setting("VERIFYGNCLOUD_CLASSES_INPUT", "classes") or "classes"
    timeout = int(get_deployment_setting("VERIFYGNCLOUD_TIMEOUT_SEC", "180") or "180")

    client = get_verifygncloud_client(api_url, api_key)
    if not hasattr(client, "run_workflow"):
        installed_version = get_inference_sdk_version()
        raise RuntimeError(
            "VERIFYGNCLOUD inference failed because the installed `inference-sdk` package is too old for workflow calls. "
            f"Detected version: {installed_version}. Upgrade with `pip install --upgrade \"inference-sdk>=1.3.2\"` "
            "and restart the app."
        )
    for attr_name in ("timeout", "request_timeout"):
        if hasattr(client, attr_name):
            try:
                setattr(client, attr_name, timeout)
            except Exception:
                pass
    classes = [spec.prompt for spec in prompt_specs]
    try:
        try:
            return client.run_workflow(
                workspace_name=workspace,
                workflow_id=workflow_id,
                images={image_input: str(image_path)},
                parameters={classes_input: classes},
                use_cache=False,
            )
        except TypeError:
            return client.run_workflow(
                workspace_name=workspace,
                workflow_id=workflow_id,
                images={image_input: str(image_path)},
                parameters={classes_input: classes},
            )
    except Exception as exc:
        message = str(exc)
        if "timeout" in message.lower():
            message = f"{message} (configured timeout: {timeout} seconds)"
        raise RuntimeError(f"VERIFYGNCLOUD inference failed: {message}") from exc


def build_hf_resolve_url(repo_id: str, filename: str, revision: str) -> str:
    normalized_repo = repo_id.strip().strip("/")
    filename_posix = str(filename).replace("\\", "/").strip("/")
    encoded_filename = "/".join(
        quote(part, safe="") for part in PurePosixPath(filename_posix).parts if part not in ("", ".")
    )
    if normalized_repo.startswith("buckets/"):
        return f"https://huggingface.co/{normalized_repo}/resolve/{encoded_filename}?download=true"
    normalized_revision = revision.strip() or "main"
    return f"https://huggingface.co/{normalized_repo}/resolve/{quote(normalized_revision, safe='')}/{encoded_filename}"


def get_configured_model_source() -> Tuple[Optional[str], str]:
    direct_url = str(getattr(config, "MODEL_DOWNLOAD_URL", "") or "").strip()
    if direct_url:
        return direct_url, "configured MODEL_DOWNLOAD_URL"
    direct_url = get_streamlit_secret("MODEL_DOWNLOAD_URL")
    if direct_url:
        return direct_url, "Streamlit Secret MODEL_DOWNLOAD_URL"

    repo_id = str(getattr(config, "MODEL_HF_REPO_ID", "") or "").strip()
    filename = str(getattr(config, "MODEL_HF_FILENAME", "") or "").strip()
    revision = str(getattr(config, "MODEL_HF_REVISION", "main") or "main").strip() or "main"
    if not repo_id:
        repo_id = get_streamlit_secret("MODEL_HF_REPO_ID")
    if not filename:
        filename = get_streamlit_secret("MODEL_HF_FILENAME")
    secret_revision = get_streamlit_secret("MODEL_HF_REVISION")
    if secret_revision:
        revision = secret_revision
    if repo_id and filename:
        return build_hf_resolve_url(repo_id, filename, revision), f"Hugging Face repo {repo_id} ({filename} @ {revision})"

    return None, ""


def get_model_request_headers() -> Dict[str, str]:
    headers = {"User-Agent": "verifygn-sam3-streamlit/1.0"}
    token_env_name = str(getattr(config, "MODEL_HF_TOKEN_ENV", "HF_TOKEN") or "HF_TOKEN").strip() or "HF_TOKEN"
    token = (
        os.getenv(token_env_name)
        or get_streamlit_secret(token_env_name)
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or get_streamlit_secret("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown size"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def bytes_from_gb(value: float) -> int:
    return int(float(value) * 1024 * 1024 * 1024)


def get_total_memory_bytes() -> Optional[int]:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int):
                return pages * page_size
        except (OSError, ValueError):
            pass
    return None


def get_remote_model_size(model_url: str) -> Optional[int]:
    request = Request(model_url, headers=get_model_request_headers(), method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                return int(content_length)
    except Exception:
        return None
    return None


def resource_guard_enabled() -> bool:
    return bool(getattr(config, "MODEL_RESOURCE_GUARD_ENABLED", True))


def has_enough_resources_for_model(model_size_bytes: Optional[int], stage: str = "load") -> bool:
    if not resource_guard_enabled() or not model_size_bytes or model_size_bytes <= 0:
        return True

    model_path = Path(config.MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    disk_free = shutil.disk_usage(model_path.parent).free
    min_free_disk = bytes_from_gb(float(getattr(config, "MODEL_MIN_FREE_DISK_GB", 1.0)))
    if stage == "download" and model_size_bytes + min_free_disk > disk_free:
        st.error("This Streamlit server does not have enough free disk space for the VFGN model.")
        st.caption(f"Model size: {format_bytes(model_size_bytes)}")
        st.caption(f"Free disk: {format_bytes(disk_free)}")
        return False

    total_memory = get_total_memory_bytes()
    ram_multiplier = max(1.0, float(getattr(config, "MODEL_LOAD_RAM_MULTIPLIER", 2.5)))
    estimated_required = int(model_size_bytes * ram_multiplier)
    if total_memory is not None and estimated_required > total_memory:
        st.error("This Streamlit Cloud machine is too small to load this VFGN model reliably.")
        st.caption(f"Model file: {format_bytes(model_size_bytes)}")
        st.caption(f"Estimated RAM needed: {format_bytes(estimated_required)}")
        st.caption(f"Server RAM: {format_bytes(total_memory)}")
        st.caption("Use a GPU/large-memory host, a Hugging Face Space with GPU, or a smaller model for cloud deployment.")
        return False

    return True


def render_model_setup_error(model_path: Path):
    st.error(f"Model file not found: {model_path}")
    st.markdown(
        """
        Streamlit Cloud cannot access the model file from your local PC.

        Configure one of these in `config.py` or in Streamlit Cloud `Secrets`:

        - `MODEL_DOWNLOAD_URL`
        - `MODEL_HF_REPO_ID` and `MODEL_HF_FILENAME`

        Streamlit Cloud example:

        ```toml
        MODEL_HF_REPO_ID = "your-name/your-model-repo"
        MODEL_HF_FILENAME = "sam3.pt"
        MODEL_HF_REVISION = "main"
        ```
        """
    )
    st.caption("For Hugging Face Storage Buckets, use `MODEL_HF_REPO_ID = \"buckets/owner/bucket-name\"` and keep `MODEL_HF_FILENAME = \"sam3.pt\"`.")
    st.caption("If your Hugging Face repo is private, also add `HF_TOKEN` in Streamlit Secrets.")
    st.caption("Very large model files may still be difficult to run on Streamlit Cloud even after download is configured.")


def download_model_file(model_path: Path, model_url: str, popup=None):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = model_path.with_suffix(model_path.suffix + ".download")
    if partial_path.exists():
        partial_path.unlink(missing_ok=True)

    timeout_sec = max(30, int(getattr(config, "MODEL_DOWNLOAD_TIMEOUT_SEC", 600)))
    request = Request(model_url, headers=get_model_request_headers())
    with urlopen(request, timeout=timeout_sec) as response, open(partial_path, "wb") as output_file:
        total_bytes = None
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            total_bytes = int(content_length)

        downloaded = 0
        last_update = 0.0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if popup is not None and (now - last_update >= 1.0 or downloaded == total_bytes):
                if total_bytes:
                    percent = min(100.0, downloaded * 100.0 / total_bytes)
                    message = f"Downloading VFGN model: {format_bytes(downloaded)} of {format_bytes(total_bytes)} ({percent:.1f}%)."
                else:
                    message = f"Downloading VFGN model: {format_bytes(downloaded)} downloaded."
                update_processing_popup(popup, "VERIFYGN MODEL DOWNLOAD", message)
                last_update = now

    if not partial_path.exists() or partial_path.stat().st_size <= 0:
        raise RuntimeError("Downloaded model file is empty.")
    partial_path.replace(model_path)


def ensure_model_available(popup=None) -> Optional[Path]:
    model_path = Path(config.MODEL_PATH)
    if model_path.exists() and model_path.stat().st_size > 0:
        if not has_enough_resources_for_model(model_path.stat().st_size, stage="load"):
            return None
        return model_path

    model_url, source_label = get_configured_model_source()
    if not model_url:
        render_model_setup_error(model_path)
        return None

    remote_model_size = get_remote_model_size(model_url)
    if remote_model_size and not has_enough_resources_for_model(remote_model_size, stage="download"):
        return None

    with MODEL_DOWNLOAD_LOCK:
        if model_path.exists() and model_path.stat().st_size > 0:
            if not has_enough_resources_for_model(model_path.stat().st_size, stage="load"):
                return None
            return model_path
        try:
            if popup is not None:
                update_processing_popup(
                    popup,
                    "VERIFYGN MODEL DOWNLOAD",
                    "Downloading VFGN model from Hugging Face. First startup can take several minutes on Streamlit Cloud.",
                )
            download_model_file(model_path, model_url, popup=popup)
        except HTTPError as exc:
            st.error(f"Model download failed with HTTP {exc.code}.")
            st.caption(f"Configured source: {source_label}")
            st.caption(str(exc))
            return None
        except URLError as exc:
            st.error("Model download failed because the app could not reach the configured source.")
            st.caption(f"Configured source: {source_label}")
            st.caption(str(exc.reason))
            return None
        except Exception as exc:
            st.error("Model download failed.")
            st.caption(f"Configured source: {source_label}")
            st.caption(str(exc))
            return None

    if model_path.exists() and model_path.stat().st_size > 0:
        if not has_enough_resources_for_model(model_path.stat().st_size, stage="load"):
            return None
        return model_path
    return None


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
    if not ensure_cv2_ready():
        return None
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
    if not ensure_cv2_ready():
        raise RuntimeError("OpenCV is not available.")
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
    if not ensure_cv2_ready():
        return None
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
    if not ensure_cv2_ready():
        raise RuntimeError("OpenCV is not available.")
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


def update_processing_popup(popup, title: str, message: str):
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    popup.markdown(
        f"""
        <div class="verifygn-overlay">
            <div class="verifygn-popup">
                <div class="verifygn-spinner"></div>
                <h3>{safe_title}</h3>
                <p>{safe_message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_processing_popup(
    title: str = "VERIFYGN DETECTION SOFTWARE",
    message: str = "VERIFYGN DETECTION SOFTWARE IS PROCESS YOUR UPLOADED FILE SO PLEASE WAIT A MINUTE WILL BE SHOWN THE RESULT",
):
    popup = st.empty()
    update_processing_popup(popup, title, message)
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
    prompt_specs = build_prompt_specs()
    if not prompt_specs:
        st.error("No prompt/class mappings are configured. Update PROMPT_CLASS_MAP in config.py.")
        return

    raw_rgb = payload.get("raw_rgb")
    if not payload.get("kind") or raw_rgb is None:
        st.error("Add an image file or camera capture before detection.")
        return

    temp_path = None
    popup = show_processing_popup()
    try:
        if payload["kind"] == "uploaded_file":
            temp_path = save_uploaded_to_temp(payload["file"])
        else:
            temp_path = save_rgb_to_temp(raw_rgb)

        backend_name = selected_inference_backend()
        legacy_backend_name = config.legacy_name("VERIFYGNCLOUD").lower()
        if backend_name in {"verifygncloud", legacy_backend_name}:
            update_processing_popup(
                popup,
                "VERIFYGN API DETECTION",
                "Sending your uploaded file to VERIFYGNCLOUD. Please wait, the result will be shown shortly.",
            )
            workflow_result = run_verifygncloud_workflow(temp_path, prompt_specs)
            predictions_output_name = get_deployment_setting("VERIFYGNCLOUD_PREDICTIONS_OUTPUT", "predictions")
            annotated_output_name = get_deployment_setting("VERIFYGNCLOUD_ANNOTATED_OUTPUT", "annotated_image")
            predictions_payload = extract_named_output(workflow_result, predictions_output_name) or workflow_result
            anns = annotations_from_verifygncloud_predictions(predictions_payload, prompt_specs, raw_rgb.shape[:2])
            # Always prefer local rendering when we have parsed detections so the
            # image shows the configured class label instead of the raw prompt.
            if anns:
                detected_rgb = draw_detections(str(temp_path), anns)
            else:
                detected_rgb = decode_image_payload(extract_named_output(workflow_result, annotated_output_name))
                if detected_rgb is None:
                    detected_rgb = draw_detections(str(temp_path), anns)

            st.session_state["sam3_result"] = {
                "name": str(payload.get("name") or "input_image"),
                "raw_image": raw_rgb,
                "detected_image": detected_rgb,
                "counts_df": counts_dataframe(anns),
                "total_count": len(anns),
                "unique_classes": len({ann.label for ann in anns}),
            }
            return

        if backend_name not in {"local", "model", "yoloe", "sam3"}:
            st.error(f"Unknown INFERENCE_BACKEND `{backend_name}`. Use `verifygncloud` or `local` in config.py.")
            return

        model_path = ensure_model_available(popup=popup)
        if model_path is None:
            return

        update_processing_popup(
            popup,
            "VERIFYGN MODEL LOADING",
            "Loading VFGN model into memory. On Streamlit Cloud CPU this can take a few minutes after the download finishes.",
        )
        backend = get_backend(
            str(model_path),
            config.HARD_CODED_CONFIDENCE,
            config.HARD_CODED_IOU,
            config.USE_FP16,
            config.HARD_CODED_IMGSZ,
        )
        update_processing_popup(
            popup,
            "VERIFYGN DETECTION SOFTWARE",
            "VERIFYGN DETECTION SOFTWARE IS PROCESS YOUR UPLOADED FILE SO PLEASE WAIT A MINUTE WILL BE SHOWN THE RESULT",
        )
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
    if not ensure_cv2_ready():
        return

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
