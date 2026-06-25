from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_imgsz(value: int, stride: int = 14) -> int:
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


class CallableTokenizerProxy:
    """Compatibility wrapper for tokenizer objects that expose encode() but are not callable."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    @staticmethod
    def _special_token_id(tokenizer, attr_names: Sequence[str], encoder_keys: Sequence[str]) -> Optional[int]:
        for name in attr_names:
            value = getattr(tokenizer, name, None)
            if isinstance(value, int):
                return value
        encoder = getattr(tokenizer, "encoder", None) or {}
        for key in encoder_keys:
            value = encoder.get(key)
            if isinstance(value, int):
                return value
        return None

    def __call__(self, texts, context_length: Optional[int] = None, truncate: bool = False):
        import torch

        if isinstance(texts, str):
            texts = [texts]
        elif texts is None:
            texts = [""]
        else:
            texts = list(texts)

        if hasattr(self._tokenizer, "tokenize"):
            try:
                return self._tokenizer.tokenize(texts, context_length=context_length)
            except TypeError:
                pass

        if not hasattr(self._tokenizer, "encode"):
            raise TypeError(f"{type(self._tokenizer).__name__} is not callable and has no encode() method.")

        context_length = int(context_length or getattr(self._tokenizer, "context_length", 77) or 77)
        start_token = self._special_token_id(
            self._tokenizer,
            ("sot_token_id", "bos_token_id", "start_token_id"),
            ("<|startoftext|>", "<start_of_text>", "<s>", "[CLS]"),
        )
        end_token = self._special_token_id(
            self._tokenizer,
            ("eot_token_id", "eos_token_id", "end_token_id"),
            ("<|endoftext|>", "<end_of_text>", "</s>", "[SEP]"),
        )

        out = torch.zeros((len(texts), context_length), dtype=torch.long)
        for row, text in enumerate(texts):
            token_ids = list(self._tokenizer.encode(text or ""))
            if start_token is not None and (not token_ids or token_ids[0] != start_token):
                token_ids = [start_token] + token_ids
            if end_token is not None and (not token_ids or token_ids[-1] != end_token):
                token_ids = token_ids + [end_token]
            if len(token_ids) > context_length:
                token_ids = token_ids[:context_length]
                if truncate and end_token is not None:
                    token_ids[-1] = end_token
            if token_ids:
                out[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        return out


class SAM3TextBackend:
    def __init__(self, model_path: str, conf: float, iou: float, half: bool, imgsz: int):
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.half = bool(half)
        self.imgsz = normalize_imgsz(imgsz)
        self.semantic_predictor = None
        self.lock = threading.Lock()

    def load(self):
        if self.semantic_predictor is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=float(self.conf),
            iou=float(self.iou),
            task="segment",
            mode="predict",
            model=self.model_path,
            half=bool(self.half),
            imgsz=int(self.imgsz),
            verbose=False,
            save=False,
        )
        self.semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
        self._patch_text_tokenizers()

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

    @staticmethod
    def _is_walkable_object(value) -> bool:
        return isinstance(value, (dict, list, tuple, set)) or hasattr(value, "__dict__")

    def _patch_text_tokenizers(self) -> int:
        predictor_model = getattr(self.semantic_predictor, "model", None)
        if predictor_model is None:
            return 0

        patched = 0
        seen = set()
        stack = [(predictor_model, 0)]
        while stack:
            current, depth = stack.pop()
            obj_id = id(current)
            if obj_id in seen or depth > 12:
                continue
            seen.add(obj_id)

            if isinstance(current, dict):
                children = list(current.values())
                attrs = []
            elif isinstance(current, (list, tuple, set)):
                children = list(current)
                attrs = []
            else:
                children = []
                try:
                    attrs = list(vars(current).items())
                except TypeError:
                    attrs = []

            for name, value in attrs:
                if "tokenizer" in name.lower() and value is not None and not callable(value):
                    if not isinstance(value, CallableTokenizerProxy) and (
                        hasattr(value, "encode") or hasattr(value, "tokenize") or hasattr(value, "encoder")
                    ):
                        setattr(current, name, CallableTokenizerProxy(value))
                        patched += 1
                        value = getattr(current, name)
                if depth < 12 and self._is_walkable_object(value):
                    children.append(value)

            if depth < 12:
                for child in children:
                    if self._is_walkable_object(child):
                        stack.append((child, depth + 1))

        return patched

    def _run_text_prompt(self, prompt: str):
        self._patch_text_tokenizers()
        try:
            return self.semantic_predictor(text=[prompt])
        except TypeError as exc:
            message = str(exc)
            if "SimpleTokenizer" not in message or "not callable" not in message:
                raise
            if self._patch_text_tokenizers() == 0:
                raise RuntimeError(
                    "SAM3 text prompting failed because Ultralytics exposed a non-callable tokenizer. "
                    "Update or reinstall ultralytics."
                ) from exc
            return self.semantic_predictor(text=[prompt])

    def run_text(self, image_path: str, prompts: Sequence[object], max_instances: int = 9999) -> List[Annotation]:
        self.load()
        prompt_specs = self._normalize_prompt_specs(prompts)
        if not prompt_specs:
            raise ValueError("No prompts provided.")
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        shape_hw = image.shape[:2]
        anns: List[Annotation] = []
        ann_id = 1
        with self.lock:
            self.semantic_predictor.set_image(image_path)
            for color_idx, spec in enumerate(prompt_specs):
                raw = self._run_text_prompt(spec.prompt)
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
