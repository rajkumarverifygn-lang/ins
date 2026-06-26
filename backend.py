from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not available. Install `opencv-python-headless>=4.8` for Streamlit Cloud "
            "or `opencv-python>=4.8` on Windows."
        ) from CV2_IMPORT_ERROR


def normalize_imgsz(value: int, stride: int = 32) -> int:
    value = max(int(stride), int(value))
    rounded = int(stride * round(value / float(stride)))
    return max(int(stride), rounded)


def as_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def binary_mask(mask: np.ndarray, shape_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            m = m.max(axis=0) if m.shape[0] < 8 else m.max(axis=-1)
    m = m > 0.5
    if shape_hw is not None and m.shape[:2] != shape_hw:
        require_cv2()
        m = cv2.resize(m.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m.astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter / union) if union > 0 else 0.0


@dataclass
class PromptSpec:
    prompt: str
    label: str


@dataclass
class Annotation:
    id: int
    label: str
    score: float
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    color_index: int = 0
    meta: Dict = field(default_factory=dict)

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mask))

    def clone(self, ann_id: Optional[int] = None, color_index: Optional[int] = None) -> "Annotation":
        return Annotation(
            id=self.id if ann_id is None else ann_id,
            label=self.label,
            score=self.score,
            mask=self.mask.copy(),
            bbox=tuple(self.bbox),
            color_index=self.color_index if color_index is None else color_index,
            meta=dict(self.meta),
        )

    @classmethod
    def from_mask(cls, ann_id: int, label: str, score: float, mask: np.ndarray, color_index: int = 0) -> "Annotation":
        m = binary_mask(mask)
        return cls(
            id=ann_id,
            label=label,
            score=float(score) if score is not None else 0.0,
            mask=m,
            bbox=bbox_from_mask(m),
            color_index=color_index,
        )


class SAM3TextBackend:
    """Text-prompted detector backend.

    This now wraps the lightweight Ultralytics **YOLOE** segmentation model
    (e.g. ``yoloe-11s-seg.pt``, ~28 MB) instead of SAM3 (~3.2 GB), so it loads
    and runs on Streamlit Cloud CPU. The class name is kept unchanged so the
    existing ``app.py`` import (`SAM3TextBackend`) keeps working.
    """

    def __init__(self, model_path: str, conf: float, iou: float, half: bool, imgsz: int):
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.half = bool(half)
        self.imgsz = normalize_imgsz(imgsz, stride=32)
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        try:
            from ultralytics import YOLOE
        except (ModuleNotFoundError, ImportError) as exc:
            self._raise_missing_dependency(exc)
        self.model = YOLOE(self.model_path)

    @staticmethod
    def _raise_missing_dependency(exc: Exception):
        missing = getattr(exc, "name", "") or str(exc)
        raise RuntimeError(
            f"Missing Python package `{missing}` required by the VFGN model. "
            "Update requirements.txt, push to GitHub, and reboot the Streamlit app."
        ) from exc

    @staticmethod
    def _normalize_prompt_specs(prompts: Sequence[object]) -> List[PromptSpec]:
        specs: List[PromptSpec] = []
        for item in prompts:
            if isinstance(item, PromptSpec):
                prompt = item.prompt.strip()
                label = item.label.strip() or prompt
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                prompt = str(item[0]).strip()
                label = str(item[1]).strip() or prompt
            else:
                prompt = str(item).strip()
                label = prompt
            if prompt:
                specs.append(PromptSpec(prompt=prompt, label=label))
        return specs

    def _set_text_classes(self, names: Sequence[str]):
        names = list(names)
        # Newer Ultralytics accepts set_classes(names) directly; older versions
        # require the text prompt embeddings from get_text_pe(names).
        try:
            self.model.set_classes(names, self.model.get_text_pe(names))
            return
        except (TypeError, AttributeError):
            pass
        self.model.set_classes(names)

    def _run_text_prompt(self, prompt: str, image_path: str):
        self._set_text_classes([prompt])
        return self.model.predict(
            image_path,
            conf=float(self.conf),
            iou=float(self.iou),
            imgsz=int(self.imgsz),
            half=False,  # fp16 is GPU-only; Streamlit Cloud runs on CPU.
            retina_masks=True,
            verbose=False,
            save=False,
        )

    def run_text(self, image_path: str, prompts: Sequence[object], max_instances: int = 9999) -> List[Annotation]:
        self.load()
        prompt_specs = self._normalize_prompt_specs(prompts)
        if not prompt_specs:
            raise ValueError("No prompts provided.")
        require_cv2()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        shape_hw = image.shape[:2]
        anns: List[Annotation] = []
        ann_id = 1
        with self.lock:
            for color_idx, spec in enumerate(prompt_specs):
                raw = self._run_text_prompt(spec.prompt, image_path)
                items = self._extract(
                    raw,
                    default_label=spec.label,
                    shape_hw=shape_hw,
                    start_id=ann_id,
                    color_offset=color_idx,
                )
                for ann in items:
                    if ann.area_px <= 0:
                        continue
                    # Force the display label to the configured class name
                    # (the model's class name is the raw text prompt).
                    ann.label = spec.label
                    anns.append(ann)
                    ann_id += 1
                    if len(anns) >= max_instances:
                        break
        return self._apply_thresholds(anns)

    def _apply_thresholds(self, anns: Sequence[Annotation]) -> List[Annotation]:
        candidates = [
            ann.clone()
            for ann in anns
            if ann.area_px > 0 and (ann.score <= 0 or ann.score >= float(self.conf))
        ]
        candidates.sort(key=lambda ann: (ann.score, ann.area_px), reverse=True)

        kept: List[Annotation] = []
        for ann in candidates:
            suppress = False
            for existing in kept:
                if existing.label != ann.label:
                    continue
                if mask_iou(existing.mask, ann.mask) >= float(self.iou):
                    suppress = True
                    break
            if not suppress:
                kept.append(ann)

        return [ann.clone(ann_id=idx) for idx, ann in enumerate(kept, start=1)]

    def _extract(self, raw, default_label: str, shape_hw: Tuple[int, int], start_id: int, color_offset: int = 0) -> List[Annotation]:
        anns: List[Annotation] = []

        if isinstance(raw, dict) and "masks" in raw:
            masks = as_numpy(raw.get("masks"))
            scores = as_numpy(raw.get("scores"))
            return self._extract_arrays(masks, scores, default_label, shape_hw, start_id, color_offset)

        if not isinstance(raw, (list, tuple)):
            raw = [raw]

        for result in raw:
            if isinstance(result, (list, tuple)) and len(result) >= 1 and not hasattr(result, "masks"):
                masks = as_numpy(result[0])
                scores = None
                anns.extend(self._extract_arrays(masks, scores, default_label, shape_hw, start_id + len(anns), color_offset))
                continue

            masks_obj = getattr(result, "masks", None)
            boxes_obj = getattr(result, "boxes", None)
            masks = None
            scores = None
            cls_ids = None
            names = getattr(result, "names", None) or {}

            if masks_obj is not None:
                masks = as_numpy(getattr(masks_obj, "data", masks_obj))
            if boxes_obj is not None:
                scores = as_numpy(getattr(boxes_obj, "conf", None))
                cls_ids = as_numpy(getattr(boxes_obj, "cls", None))

            if masks is None:
                continue
            arr_anns = self._extract_arrays(masks, scores, default_label, shape_hw, start_id + len(anns), color_offset)
            if cls_ids is not None and len(cls_ids) == len(arr_anns):
                for ann, cls_id in zip(arr_anns, cls_ids):
                    try:
                        ann.label = str(names.get(int(cls_id), default_label))
                    except Exception:
                        ann.label = default_label
            anns.extend(arr_anns)
        return anns

    def _extract_arrays(
        self,
        masks,
        scores,
        default_label: str,
        shape_hw: Tuple[int, int],
        start_id: int,
        color_offset: int,
    ) -> List[Annotation]:
        if masks is None:
            return []
        m = as_numpy(masks)
        if m is None:
            return []
        if m.ndim == 2:
            m = m[None, ...]
        if m.ndim == 3 and m.shape[0] == shape_hw[0] and m.shape[1] == shape_hw[1] and m.shape[2] < 512:
            m = np.moveaxis(m, -1, 0)
        scores_np = as_numpy(scores)
        anns: List[Annotation] = []
        for i in range(m.shape[0]):
            score = float(scores_np[i]) if scores_np is not None and len(scores_np) > i else 0.0
            mask = binary_mask(m[i], shape_hw)
            if np.count_nonzero(mask) == 0:
                continue
            anns.append(Annotation.from_mask(start_id + len(anns), default_label, score, mask, color_index=color_offset + i))
        return anns
