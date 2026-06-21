"""Visual pipeline — Phase B.

Isolates the bike from the detail-page image with YOLOv8 (detection) + SAM
(segmentation), replaces the background with white, tight-crops, and returns the
result for CLIP embedding into ``offer_features.clip_vector_visual``.

Models auto-download on first use (cached in ~/.cache):
  * yolov8n.pt  (~6 MB)  — detection, COCO class 1 = bicycle
  * sam2_b.pt   (~150 MB)— segmentation (override with SEGMENTATION_SAM_MODEL)

Everything degrades gracefully: if SAM can't load or doesn't return a usable
mask, we fall back to the YOLO bbox crop, then to the full image. Download
failures leave ``clip_vector_visual`` NULL.
"""
import io
import logging
import os
import sqlite3

import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)

_YOLO = None
_SAM = "unloaded"  # sentinel distinct from None (None = tried and failed)

YOLO_MODEL = os.environ.get("SEGMENTATION_YOLO_MODEL", "yolov8n.pt")
SAM_MODEL = os.environ.get("SEGMENTATION_SAM_MODEL", "sam2_b.pt")
_BICYCLE_CLASS = 1  # COCO


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(YOLO_MODEL)
    return _YOLO


def _get_sam():
    """Load SAM lazily; returns None (and logs once) if it can't be loaded."""
    global _SAM
    if _SAM == "unloaded":
        try:
            from ultralytics import SAM
            _SAM = SAM(SAM_MODEL)
        except Exception as exc:
            log.warning("SAM unavailable (%s) — falling back to bbox crops", exc)
            _SAM = None
    return _SAM


def download_image(url: str, timeout: int = 15) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        log.warning("Image download failed for %s: %s", url, exc)
        return None


def crop_to_bbox(image: Image.Image, bbox: list[float], padding: float = 0.03) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = image.size
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    box = (max(0, int(x1) - pad_x), max(0, int(y1) - pad_y),
           min(w, int(x2) + pad_x), min(h, int(y2) + pad_y))
    return image.crop(box)


def _extract_on_white(image: Image.Image, mask: np.ndarray, padding: float = 0.03) -> Image.Image:
    """Composite the masked pixels onto white, then tight-crop to the mask bbox."""
    img_array = np.array(image.convert("RGB"))
    output = np.full_like(img_array, 255)
    output[mask] = img_array[mask]
    result = Image.fromarray(output)

    ys, xs = np.where(mask)
    # +1 on the max edges: mask indices are inclusive, PIL crop edges exclusive.
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return crop_to_bbox(result, bbox, padding)


def filter_min_max(img, size, mode='dilation'):
        """Find the local min or max in a sliding window."""
        from numpy.lib.stride_tricks import sliding_window_view
        pad_w = size // 2
        
        if len(img.shape) == 3:
            pad_width = ((pad_w, pad_w), (pad_w, pad_w), (0, 0))
            axis = (0, 1)
        else:
            pad_width = ((pad_w, pad_w), (pad_w, pad_w))
            axis = (0, 1)

        pad_img = np.pad(img, pad_width, mode='edge')
        windows = sliding_window_view(pad_img, window_shape=(size, size), axis=axis)

        if mode == 'dilation':
            return np.max(windows, axis=(-2, -1))
        else: # erosion
            return np.min(windows, axis=(-2, -1))


def segment_with_steps(image: Image.Image, conf_threshold: float = 0.5) -> dict:
    """Run the YOLO + SAM pipeline and return every intermediate artifact.

    Unlike :func:`segment_bike`, this keeps the bbox *and* the raw SAM mask so
    callers (e.g. the dashboard showcase) can visualise each step.

    Returns {success, bbox, mask, crop, reason}: ``bbox`` is the YOLO detection
    box (None if nothing detected), ``mask`` the boolean SAM mask (None if SAM
    produced none), ``crop`` the final white-bg PIL crop (None unless reason ==
    'ok'). ``reason`` is 'ok' / 'no_detection' / 'no_mask' / 'mask_too_small'.
    """
    yolo = _get_yolo()
    results = yolo(image, classes=[_BICYCLE_CLASS], verbose=False)
    boxes = [b for b in results[0].boxes if float(b.conf) > conf_threshold]
    if not boxes:
        return {"success": False, "bbox": None, "mask": None, "crop": None, "reason": "no_detection"}

    best = max(boxes, key=lambda b: float(b.conf))
    bbox = best.xyxy[0].tolist()

    sam = _get_sam()
    if sam is None:
        return {"success": False, "bbox": bbox, "mask": None, "crop": None, "reason": "no_mask"}

    sam_results = sam(image, bboxes=[bbox], verbose=False)
    masks = sam_results[0].masks
    if masks is None or len(masks.data) == 0:
        return {"success": False, "bbox": bbox, "mask": None, "crop": None, "reason": "no_mask"}

    mask = masks.data[0].cpu().numpy().astype(bool)

    # filling in the small holes in the mask
    mask = filter_min_max(mask.astype(np.uint8), size=5, mode='dilation').astype(bool)


    img_area = image.size[0] * image.size[1]
    mask_area = int(mask.sum())
    if mask_area < img_area * 0.02 or mask_area > img_area * 0.98:
        return {"success": False, "bbox": bbox, "mask": mask, "crop": None, "reason": "mask_too_small"}

    crop = _extract_on_white(image, mask)
    return {"success": True, "bbox": bbox, "mask": mask, "crop": crop, "reason": "ok"}


def segment_bike(image: Image.Image, conf_threshold: float = 0.5) -> dict:
    """Detect + segment the bike.

    Returns {success, segmented_image, bbox, reason} where reason is one of
    'ok' / 'no_detection' / 'no_mask' / 'mask_too_small'.
    """
    steps = segment_with_steps(image, conf_threshold)
    return {
        "success": steps["success"],
        "segmented_image": steps["crop"],
        "bbox": steps["bbox"],
        "reason": steps["reason"],
    }


def get_visual_source_url(conn: sqlite3.Connection, offer_id: str) -> str | None:
    """The image URL for the visual pipeline: first full-definition gallery image
    (is_thumbnail=0), falling back to the listing thumbnail."""
    row = conn.execute(
        "SELECT url FROM offer_images WHERE offer_id = ? AND is_thumbnail = 0 "
        "ORDER BY position ASC LIMIT 1",
        (offer_id,),
    ).fetchone()
    if row and row[0]:
        return row[0]
    row = conn.execute("SELECT thumbnail_url FROM offers WHERE id = ?", (offer_id,)).fetchone()
    return row[0] if row else None
