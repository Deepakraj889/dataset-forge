from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import numpy as np

def detect_solid_borders(
    img: np.ndarray,
    tolerance: float = 8.0,
    max_border_ratio: float = 0.15
) -> Tuple[int, int, int, int]:
    """
    Detects solid letterbox/banner bars (e.g. black, white, or uniform bars) at image edges.
    Returns: (crop_top, crop_bottom, crop_left, crop_right) in pixels to trim.
    """
    h, w = img.shape[:2]
    max_h_trim = int(h * max_border_ratio)
    max_w_trim = int(w * max_border_ratio)

    if max_h_trim < 2 or max_w_trim < 2:
        return (0, 0, 0, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    center_mean = float(np.mean(gray[cy1:cy2, cx1:cx2]))

    def is_border_bar(strip: np.ndarray) -> bool:
        if strip.size == 0:
            return False
        strip_std = float(np.std(strip))
        if strip_std > tolerance:
            return False
        strip_mean = float(np.mean(strip))
        if strip_mean < 25.0 or strip_mean > 240.0:
            return True
        return abs(strip_mean - center_mean) > 30.0

    top_trim = 0
    if is_border_bar(gray[0:2, :]):
        ref_top = gray[0, :]
        for y in range(1, max_h_trim):
            row = gray[y:y+1, :]
            if np.std(row - ref_top) > tolerance or not is_border_bar(row):
                break
            top_trim = y

    bottom_trim = 0
    if is_border_bar(gray[h-2:h, :]):
        ref_bottom = gray[h-1, :]
        for y in range(1, max_h_trim):
            row = gray[h-y-1:h-y, :]
            if np.std(row - ref_bottom) > tolerance or not is_border_bar(row):
                break
            bottom_trim = y

    left_trim = 0
    if is_border_bar(gray[:, 0:2]):
        ref_left = gray[:, 0]
        for x in range(1, max_w_trim):
            col = gray[:, x:x+1]
            if np.std(col - ref_left) > tolerance or not is_border_bar(col):
                break
            left_trim = x

    right_trim = 0
    if is_border_bar(gray[:, w-2:w]):
        ref_right = gray[:, w-1]
        for x in range(1, max_w_trim):
            col = gray[:, w-x-1:w-x]
            if np.std(col - ref_right) > tolerance or not is_border_bar(col):
                break
            right_trim = x

    return (top_trim, bottom_trim, left_trim, right_trim)

def generate_corner_mask(
    img: np.ndarray,
    region: str = "corners",
    margin_ratio: Optional[float] = None,
    margin_ratio_h: float = 0.12,
    margin_ratio_w: float = 0.20,
    sensitivity: int = 22
) -> np.ndarray:
    """
    Resolution-adaptive detection of text, signatures, and watermarks across target margins.
    Uses Top-Hat (bright on dark) + Black-Hat (dark on bright) morphology to capture the
    full solid body of letters/logos rather than just hollow edges.
    """
    if margin_ratio is not None:
        margin_ratio_h = margin_ratio
        margin_ratio_w = margin_ratio

    h, w = img.shape[:2]
    scale = max(h, w) / 1024.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Scale structuring element to image resolution so thick fonts are fully captured
    k_size = max(5, int(round(12 * scale)))
    if k_size % 2 == 0:
        k_size += 1
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    # Top-hat captures bright text, Black-hat captures dark text
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel)
    contrast = cv2.max(top_hat, black_hat)

    # Threshold contrast
    _, bin_mask = cv2.threshold(contrast, sensitivity, 255, cv2.THRESH_BINARY)

    # Morphological close to bridge gaps between letter strokes
    close_k = max(3, int(round(4 * scale)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, close_kernel)

    # Dilate generously so anti-aliasing and soft letter boundaries are 100% swallowed
    dilate_k = max(3, int(round(5 * scale)))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)

    # Restrict mask to target region
    mask = np.zeros((h, w), dtype=np.uint8)
    mh = int(h * margin_ratio_h)
    mw = int(w * margin_ratio_w)

    if region == "bottom-right":
        mask[h - mh:, w - mw:] = dilated[h - mh:, w - mw:]
    elif region == "bottom":
        mask[h - mh:, :] = dilated[h - mh:, :]
    elif region == "corners":
        # All 4 corners + bottom margin strip (common for artist signatures)
        mask[:mh, :mw] = dilated[:mh, :mw]
        mask[:mh, w - mw:] = dilated[:mh, w - mw:]
        mask[h - mh:, :mw] = dilated[h - mh:, :mw]
        mask[h - mh:, w - mw:] = dilated[h - mh:, w - mw:]
        # Include narrow bottom strip (last 4% height) where signatures often sit
        bottom_strip = int(h * 0.05)
        mask[h - bottom_strip:, :] = dilated[h - bottom_strip:, :]
    elif region == "all-margins":
        mask[:mh, :] = dilated[:mh, :]
        mask[h - mh:, :] = dilated[h - mh:, :]
        mask[:, :mw] = dilated[:, :mw]
        mask[:, w - mw:] = dilated[:, w - mw:]
    else:
        mask[h - mh:, w - mw:] = dilated[h - mh:, w - mw:]

    return mask

def inpaint_image(
    img: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: Optional[int] = None,
    method: str = "telea"
) -> np.ndarray:
    """
    Inpaints masked regions using OpenCV Telea or Navier-Stokes algorithm.
    Optimized to inpaint individual bounding regions with local padding for superior
    texture blending and 10x faster execution on high-res images.
    """
    if not np.any(mask):
        return img.copy()

    h, w = img.shape[:2]
    scale = max(h, w) / 1024.0

    if inpaint_radius is None:
        inpaint_radius = max(4, int(round(4 * scale)))

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS

    # Find connected watermark components
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = img.copy()

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 9:
            continue
        pad = inpaint_radius * 2 + 8
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        sub_img = cleaned[y1:y2, x1:x2]
        sub_mask = mask[y1:y2, x1:x2]

        cleaned[y1:y2, x1:x2] = cv2.inpaint(sub_img, sub_mask, inpaintRadius=inpaint_radius, flags=flag)

    return cleaned

def remove_watermark(
    image_path: Path,
    output_path: Path,
    region: str = "corners",
    auto_trim_borders: bool = True,
    method: str = "telea",
    inpaint_radius: Optional[int] = None
) -> bool:
    """
    Cleans watermarks, signatures, and solid letterbox borders from an image.
    1. Checks and trims solid letterbox bars if detected.
    2. Identifies solid text/signature bodies in target corner/margin regions.
    3. Inpaints masked areas with surrounding texture.
    """
    try:
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return False

        h, w = img.shape[:2]

        # Step 1: Trim solid border bars if present
        if auto_trim_borders:
            top_trim, bottom_trim, left_trim, right_trim = detect_solid_borders(img)
            if top_trim > 0 or bottom_trim > 0 or left_trim > 0 or right_trim > 0:
                y1 = top_trim
                y2 = h - bottom_trim
                x1 = left_trim
                x2 = w - right_trim
                if (y2 - y1) > h * 0.5 and (x2 - x1) > w * 0.5:
                    img = img[y1:y2, x1:x2]

        # Step 2: Generate resolution-adaptive watermark mask
        mask = generate_corner_mask(img, region=region)

        # Step 3: Inpaint
        cleaned = inpaint_image(img, mask, inpaint_radius=inpaint_radius, method=method)

        # Step 4: Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            _, buf = cv2.imencode(ext, cleaned, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        else:
            _, buf = cv2.imencode(".png", cleaned)

        with open(output_path, "wb") as f:
            f.write(buf)

        return True
    except Exception:
        return False

def clean_watermark_worker(args: tuple) -> bool:
    """Multiprocessing worker function for parallel watermark cleaning."""
    image_path, output_path, region, auto_trim_borders, method, inpaint_radius = args
    success = remove_watermark(
        image_path=image_path,
        output_path=output_path,
        region=region,
        auto_trim_borders=auto_trim_borders,
        method=method,
        inpaint_radius=inpaint_radius
    )
    if success:
        txt_pair = image_path.with_suffix(".txt")
        if txt_pair.exists():
            out_txt = output_path.with_suffix(".txt")
            out_txt.write_text(txt_pair.read_text(encoding="utf-8"), encoding="utf-8")
    return success

